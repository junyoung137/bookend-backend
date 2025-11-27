--
-- PostgreSQL database dump
--

\restrict PSgo45UvujqHKhtF5BiRz7HONftQcKijvV95ApXFfh6Tm5RFozcWFGPaMumJokB

-- Dumped from database version 15.15 (Homebrew)
-- Dumped by pg_dump version 15.15 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: feedbacks; Type: TABLE; Schema: public; Owner: ace_admin
--

CREATE TABLE public.feedbacks (
    id integer NOT NULL,
    user_id text,
    original text NOT NULL,
    selected_feature text NOT NULL,
    corrected_text text NOT NULL,
    feedback text NOT NULL,
    context jsonb,
    "timestamp" timestamp without time zone NOT NULL,
    processed integer DEFAULT 0,
    processed_at timestamp without time zone
);


ALTER TABLE public.feedbacks OWNER TO ace_admin;

--
-- Name: feedbacks_id_seq; Type: SEQUENCE; Schema: public; Owner: ace_admin
--

CREATE SEQUENCE public.feedbacks_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE public.feedbacks_id_seq OWNER TO ace_admin;

--
-- Name: feedbacks_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: ace_admin
--

ALTER SEQUENCE public.feedbacks_id_seq OWNED BY public.feedbacks.id;


--
-- Name: feedbacks id; Type: DEFAULT; Schema: public; Owner: ace_admin
--

ALTER TABLE ONLY public.feedbacks ALTER COLUMN id SET DEFAULT nextval('public.feedbacks_id_seq'::regclass);


--
-- Name: feedbacks feedbacks_pkey; Type: CONSTRAINT; Schema: public; Owner: ace_admin
--

ALTER TABLE ONLY public.feedbacks
    ADD CONSTRAINT feedbacks_pkey PRIMARY KEY (id);


--
-- Name: idx_processed; Type: INDEX; Schema: public; Owner: ace_admin
--

CREATE INDEX idx_processed ON public.feedbacks USING btree (processed);


--
-- Name: idx_timestamp; Type: INDEX; Schema: public; Owner: ace_admin
--

CREATE INDEX idx_timestamp ON public.feedbacks USING btree ("timestamp");


--
-- Name: idx_user_id; Type: INDEX; Schema: public; Owner: ace_admin
--

CREATE INDEX idx_user_id ON public.feedbacks USING btree (user_id);


--
-- PostgreSQL database dump complete
--

\unrestrict PSgo45UvujqHKhtF5BiRz7HONftQcKijvV95ApXFfh6Tm5RFozcWFGPaMumJokB

